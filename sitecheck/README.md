# sitecheck

You already have a spot in mind. `sitecheck` checks whether the ground there shows traces
of people: house pits, foundations, old roads, ditches. It combines that with what you know
(finds, legends, old maps) into one JSON report and a ready prompt for any AI.

- **No built-in AI, no API costs.** All analysis is local Python.
- **Offline analysis.** Internet is used once, optionally, to download data for your spot.
- **Every number has a source:** `user`, `computed:DEM`, `computed:water.geojson`.
- **You stay in control.** Detections are *candidates* until you confirm or reject them.

## Install

```bash
pip install -e .                 # core (works for USA, England, Finland, point clouds)
pip install -e ".[geo]"          # + rasterio/pyproj: ArcticDEM, any projection, faster downloads
pip install -e ".[lidar]"        # + LAS/LAZ point clouds
```

## Interface (recommended)

```bash
sitecheck ui                      # opens http://127.0.0.1:8765 in your browser; works offline
sitecheck ui --workspace my_sites # keep areas in a folder of your choice
```
- **Areas**: import a folder (one sub-folder per area with screenshots and a coordinates .txt),
  or add an area by uploading screenshots and pasting coordinates.
- **Layers** (keys 1-4): picture 1, picture 2 (aligned automatically), relief from shading,
  soil-colour anomalies (dark/reddish = red, light = yellow, green crop = cyan).
- **Tools**:
  - Move: drag, wheel to zoom.
  - Scale: click two points and enter the distance.
  - Mask: drag a box over program buttons.
  - Spot: click your spot.
  - Add pit: click a hollow the program missed.
  - **Guided row**: click 2-3 neighbouring patches you think are houses. The program continues
    the line at the same spacing, circles further similar patches, marks expected places with
    nothing (`?`, worth checking on the ground), gives the chance that a random line does as well,
    and checks the same places in the other pictures.
- **Candidates**: confirm ✓ or reject ✗. The score and report update at once.
  "Copy AI prompt" puts the ready prompt on the clipboard.

## Command line

```bash
sitecheck ask -o context.json                                  # once: what you know
sitecheck run --point 51.7520,-1.2577 --context context.json   # download + analyze
sitecheck review site_51.75200_-1.25770 --confirm 1,2 --reject 6 --add pit:35,-20:6
```

Then paste `prompt.md` into any AI chat and attach `candidates.png`, `terrain_panels.png`
(and `overlay.png` if you used an old map).

`run` picks the best elevation source for the spot automatically and falls back to the
next one if a service is down or has no data there. Force one with `--source`.

## Elevation data by country

Pits and foundations are 3-15 m across and 0.2-1 m deep: you need **1-2 m lidar**.
The report always says which data was used and warns when it can't show small features.

### USA - automatic
USGS 3DEP 1 m bare-earth lidar, clipped on the server (`--source 3dep`).
3DEP still has gaps. In those areas the service quietly fills the window with 10 m or 30 m data
resampled to 1 m. `sitecheck` detects this ("looks resampled from a much coarser source")
instead of pretending it's lidar. In that case use a state lidar portal tile with `--dem`.

### England - automatic
Environment Agency lidar composite DTM, 1 m, about 99 % of England (`--source ea`), under the
Open Government Licence. Wales and Scotland are not in this service: download from DataMapWales
or the Scottish Remote Sensing Portal and use `--dem`.

### Finland - automatic with a free key
National Land Survey (NLS) 2 m DTM from lidar, covering the whole country (`--source nls`).
1. Register (free) at the NLS open-data API service and copy your API key.
2. Pass `--nls-key KEY`, or set `NLS_API_KEY`, or save `{"nls_api_key": "KEY"}` in `~/.sitecheck.json`.

Without a key, `run` uses ArcticDEM (north) or 30 m data and says so. NLS tiles are in
EPSG:3067, which `sitecheck` handles even without pyproj.

### Russia - no open lidar, so three routes

1. **North of ~60 N (Karelia, Leningrad and Arkhangelsk oblasts, Komi, Siberia...): ArcticDEM.**
   This is a 2 m model made from satellite stereo images (`--source arcticdem`, needs `.[geo]`).
   It is a *surface* model, so the ground filter removes trees and roofs automatically.
   Works well in fields and sparse forest. Under closed canopy the ground is guessed and the
   report warns about it. PGC lists 2 m mosaics for education/research use, while 10 m (`arcticdem10`) is
   open to all. Check which applies to you.

2. **Anywhere: make your own lidar.** This is the real answer for central and southern Russia.
   - **Drone photogrammetry:** fly a grid with any camera drone (overlap about 75 %, flight height 60-80 m)
     and process the photos in **OpenDroneMap / WebODM** (free, runs offline on your PC). Use its
     `dtm.tif` directly with `--dem`, or its `dsm.tif` with `--dem ... --dsm`. The result is 5-10 cm
     resolution, better than any national lidar. It sees bare ground in fields and meadows,
     but not under forest.
   - **Phone lidar** (iPhone/iPad Pro scanning apps), or any LAS/LAZ/PLY/XYZ point cloud:
     ```bash
     sitecheck grid scan.ply --anchor 56.1000,40.4000 --center --cell 0.25 -o scan_dem.tif
     sitecheck analyze --point 56.1000,40.4000 --dem scan_dem.tif --dem-label "phone lidar"
     ```
     Use `--anchor` for scans in local metres, or `--epsg 32637` (etc.) for UTM coordinates.
     The lowest point in each cell is taken as ground, then the ground filter removes vegetation.

3. **Old maps, the strongest evidence where relief data is weak.** PGM (General Survey),
   Mende, Schubert and RKKA 1:50 000 sheets often show villages and farmsteads that have since vanished.
   Georeference the scan once in QGIS (free; Raster → Georeferencer, save as GeoTIFF),
   then add `--overlay old_map.tif`. You get `overlay.png` with candidates over the old map.
   This works in any country, for example with Finnish Senate maps, OS 6-inch maps or USGS historical topos.

South of 60 N without your own data, `run` uses Copernicus GLO-30 (30 m). That gives landscape
context only (slope, water, landform), and the report says pits are not visible.

### No elevation data at all: screenshots (`sitecheck image`)

This is for when the only thing available is a satellite view, for example most of Russia south of 60 N.
Take one or more screenshots of the same view, ideally from **different seasons** (Google Earth has
historical imagery). Early spring with low sun, no leaves and leftover snow is the most informative.

```bash
sitecheck image summer.jpg spring.jpg --point 53.454131,50.801542 \
    --known 412,380,980,395,140 \
    --mask 1480,0,1600,300
```
- `--known X1,Y1,X2,Y2,METRES` sets the scale from two pixels and the real distance between them
  (measure it with the Google Earth ruler). Or use `--mpp`. Without a scale, 0.3 m/px is assumed and flagged.
- `--mask` hides program buttons (compass, zoom slider). `--spot-px` gives the spot's pixel (default: centre).
- `--point` puts coordinates on every candidate and settles the sun direction.

What it computes, all classical image maths:
1. **Masks**: forest (rough and dark, or very rough), cultivated fields (wide rough or very dark areas), snow.
2. **Relief from shading** (linear photoclinometry): brightness changes on grass or soil are proportional
   to the slope towards the sun. Integrating them gives a rough relative relief, and the normal pit
   detector runs on it. Confidence = how far a hollow stands out above that image's noise.
3. **Sun direction** from bush and hollow shadows. The 180° ambiguity is settled by the latitude.
4. **Marks**: snow lingering in hollows, dark or lush spots, isolated bushes.
5. **Several images** are aligned automatically by edge structure, which works across seasons. A place
   seen in two pictures, or holding snow, gets more weight. Shading and colour in the *same* picture
   are not counted twice.

Output: `candidates.png`, `candidates_image2.png`... (the same numbers on every screenshot),
`relief_from_shading.png`, `report.json`, `prompt.md`. `sitecheck review` works as usual.

How good is it? On a synthetic photo with known pits: 28/30 found, about 5 false candidates per image.
On real screenshots expect more false hits (bushes, cow paths, damp patches), so check each one.
It's a tool for finding places to walk to, not a substitute for lidar.

### Many areas at once (`sitecheck batch`)

```
russland/
  1/  1.txt  (53°00′57.00″ N, 49°30′57.16″ E)   file.webp   file (1).webp
  2/  2.txt  ...
```
`sitecheck batch russland -o results` processes every sub-folder. It reads coordinates in DMS
(also with Russian С/Ю/В/З) or decimal form, picks the sharpest image as the reference, aligns
the others (different zoom and framing are fine), and masks the imagery credit at the bottom.
It writes one result folder per area plus `summary.md` (a table) and `overview.jpg` (all areas on one sheet).

### Fields and woodland in screenshots
- **Soil and crop marks** in fields, and smaller ones in meadows: patches of darker or lighter soil
  and greener or parched crop, 7-90 m across, after smoothing away furrows and tramlines. A cluster
  of marks is what a ploughed-out village looks like. The program also measures **mottling**,
  the share of ground covered by patches. If the whole field is blotchy, natural soil (alkaline
  spots, subsidence hollows, wet patches) is more likely, and confidence is lowered.
- **Woodland**: clearings inside woods, patches of trees of a different colour (old orchards,
  lilac and poplars often mark farmsteads), straight tree rows, and sets of 3+ parallel rows
  (a planted grid: estates, streets, orchards or shelterbelts).
- **Groups of hollows**: three or more hollows within 30 m raise each other's confidence,
  because random noise doesn't cluster like house pits.

Known weak spots (from 13 test areas in the Samara and Orenburg regions):
- Isolated bushes: a bush's shadow can look like a hollow, so expect false hollows right next to bushes.
- Overgrown scrub: tree rows in scrub are only partly found.
- Blurry, low-resolution screenshots give little.

### Regular rows: "something may have stood here"

A ploughed-out village rarely leaves clear single features, but its houses stood in a line
at fairly even intervals. `sitecheck image` and `batch` look for this:
1. Many weak, roughly round patches (4-25 m) of the same kind: dark/reddish soil, light soil,
   green crop, or hollows from the shading relief.
2. Lines through 4+ similar patches with near-constant spacing (one missing house allowed).
3. Rejected: rows strung along a tramline or track, and rows that are really one continuous band
   (slope break, field margin, wet strip). Rows parallel to the cultivation lines are flagged and halved.
4. **Chance check**: the same patches are scattered at random over the same ground 80 times, and
   the program counts how often random layouts give an equally good row. The reported `chance` is
   corrected for every search made (4 kinds of patch × number of pictures). A row found again
   in another season multiplies the two chances.
5. Under 10% = "presumption", 10-30% = "weak presumption". It is a presumption, not proof.
   Natural soil spots can cluster more than random, so the chance is, if anything, optimistic.

On artificial images without a village it finds no rows. On the 13 Russian test areas it gives
weak presumptions in 4 areas and none in 1 and 6. Detection of the individual patches is
the limit, not the statistics.

### Any other data
Any single-band GeoTIFF works with `--dem` (projected or geographic). Mark surface models with `--dsm`.

## How detection works (no AI)

1. **Local Relief Model**: DEM minus a local quadratic trend surface (35 m window).
2. **Pits**: closed depressions, filtered by depth, diameter and roundness.
3. **Rectangles**: solid platforms or wall outlines with support on all sides and real corners.
4. **Linear features**: straight lines (roads, ditches) are told apart from curved natural gullies.
   Banks beside a ditch are merged into it, and lines that follow the contour are flagged.
5. **Ground filter** (surface models only): a progressive morphological filter removes trees and
   buildings while keeping low banks and mounds.
6. **Resolution check**: detects DEMs that were upsampled from coarse data.
7. **Score**: points from `weights.json`, each one explained. It is a heuristic, not a probability.

## Tests

```bash
python tests/test_offline.py      # every downloader against a mock server + no-lidar tools
python tests/test_image_mode.py   # screenshot mode on a synthetic 'photo' with known pits
```

Results:
- 18/18 checks pass.
- Clean 1 m terrain: 30/30 pits, 9/10 foundations, 10/10 roads (10 runs).
- Point cloud with 15 % vegetation returns: 3/3 pits and the foundation.
- Surface model with forest: 246 candidates without the filter, 16 with it (pits found: 0 → 3).

**Not yet tested against the live services** (the build machine has no internet): 3DEP, the
Environment Agency, NLS, ArcticDEM STAC and Overpass. They follow the published APIs, and
the NLS and ArcticDEM clients discover process and collection names from the service itself.
The first real run will show any mismatch, with a clear error message.
