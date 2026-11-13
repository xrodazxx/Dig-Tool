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
pip install -e ".[train]"        # + scikit-learn, only to retrain the spot model on your own marked areas
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

Give the scale if you can: `--altitude 450` (Google Earth eye altitude, looking straight down),
or `--mpp`, or `--known` (two points and the distance between them). In the interface: "eye
altitude" next to metres per pixel.

The eye altitude only works with the **width of the map view** in screen pixels (`--view-width`,
default 1920 = full-screen map). Google Earth Pro's left panel makes the map narrower, so every size
would come out too small. Measured once by the user (Full HD screen, panel open): a 200 m ruler line
was 219 px at 1.39 km eye altitude, i.e. **metres per pixel = eye altitude / 1515**, which is a map
view of 1749 px. Put a line `window 1749 px` in the pack's General info and it holds for the whole pack.

Measuring your own: in Google Earth Pro draw a line of known length with Tools → Ruler, screenshot,
count its pixels: `window = 2 × altitude × 0.577 × pixels / length`.

Caution: Google Earth's eye altitude is measured **above sea level**, so the camera is really
`altitude − ground height` above the ground. At 400 m with the ground at 150 m that is a 40 %
difference. The calibration above already includes the ground height of the place where it was
measured (unknown). If your places differ a lot in height, write `elev 150 m` (the "elev" Google Earth
shows at the bottom of the window) in an area's notes, or in General info for the whole pack.

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

### Rows of houses: "something may have stood here"

A ploughed-out village rarely leaves clear single features, but its houses stood in a line.
`sitecheck image` and `batch` look for this:
1. Many weak patches (4-35 m) of the same kind: dark/reddish soil, light soil, green crop, or
   hollows from the shading relief.
2. Lines through 4+ similar patches, each next one 10-75 m on, without a big hole in the line.
   Real houses are **not** evenly spaced (yards differ), so even spacing only raises the score;
   it is not required.
3. Rejected: rows strung along a tramline or track, and "rows" that are one continuous band
   (the road itself, a slope break, a wet strip): there must be ordinary ground between the patches.
   Rows parallel to the cultivation lines are flagged and halved.
4. **Chance check, two ways; the worse one counts.** (a) the same patches scattered at random
   80 times; (b) every patch moved up to 25 m from where it is, 80 times. (b) keeps the
   patchiness of the soil, so mottled ground that makes lines easily doesn't fool it. The
   reported `chance` is corrected for every search made (4 kinds of patch × pictures). A row
   found again in another year multiplies the chances.
5. Under 10% = "presumption", 10-20% = "weak presumption". It is a presumption, not proof.

**The other side of the street.** Once a row is found, the program looks for a second row of
the same kind of patch, parallel to it, 20-150 m away, with its own chance check. If it passes,
a dashed yellow "street?" line is drawn between the two rows.

Tested on the marked packs (screenshots at an assumed 0.3 m/px):

| place | result |
|---|---|
| Rus_earth field/2 (2 rows of 7 houses) | both rows, 7/8 and 7/7 patches on marked houses, seen in 3 of 5 years, chance < 0.1% |
| pack field/4 (2 rows + marked road) | both rows (6/7 and 3/4 on houses), the "street?" line lies 80% on the marked road |
| pack EmptyPlace (2 pictures, nothing ever there, very patchy soil) | no rows; score 17 and 4 (low). Version 0.7 gave a false row and 28 "moderate" |
| pack field/1, field/3, field/0, MixFoundPit/14 | no significant row: the houses there are found as patches, but not enough of them line up |

Per picture, a true row reaches chance < 15% in about 1 picture in 5. That's why several years
of the same place help so much. Patchy soil also scores lower now: colour marks seen in one
picture only give at most 4 points, marks that come back in 2+ pictures up to 10.

### Learned spot detector

Besides the hand-made detectors, screenshot mode uses a small model trained on the marked packs
(Rus_earth: 14 areas, 27 pictures; second pack: 20 areas, 70 pictures plus 2 empty places).
For every round spot it measures ~40 things (colour and brightness against the surroundings,
rim, shadow side vs sun, roughness, size, similar spots nearby, rank in this picture) and gives a
0-1 score. It re-ranks the other candidates and adds spots they missed (at half weight).
- It's 250 small decision trees in `sitecheck/model_pits.json`, run with plain numpy. No AI service,
  no internet.
- Not used on very smooth bare fields, or on pictures where nothing resembles the training
  examples (`report.json` → `learned_model`).

**Honest result.** The hard test is a pack the model never saw: train on one pack, test on the other.

| tested on (model trained on the other pack) | AP before | AP after | correct in top 30 |
|---|---|---|---|
| second pack, pit sites (58 pictures) | 0.17 | 0.20 | 24% → 26% |
| second pack, fields (12) | 0.24 | 0.23 | 42% → 38% |
| second pack, empty places: candidates scored ≥ 0.5 | 12 and 6 | 0 and 3 | |
| Rus_earth, all (27) | 0.22 | 0.24 | 25% → 31% |

AP = average precision of the ranked list (1.0 = every marked object above every false one).
Version 0.7 claimed 0.22 → 0.32. That was tested area by area **inside one pack**, and it did
not carry over to the second pack (0.18 → 0.20). On its own, the model ranks spots about as
well as plain spot strength (AP 0.19-0.25 vs 0.19-0.27). Its gain is small and comes from the
combination. Single pits in screenshots stay hard; rows over several years are the stronger signal.

**Third pack (Geo_Pack: 8 foundation sites, 1 rampart, 13 hard empty fields, real scale from eye
altitude).** Tested with a model that had never seen it:

| | before model | with model |
|---|---|---|
| AP, all marked objects (40 pictures) | 0.18 | 0.22 |
| AP, house pits | 0.12 | 0.15 |
| AP, foundations | 0.17 | 0.19 |
| found at all (any rank): foundations / pits | 77% / 49% | 82% / 57% |
| top-30 spots that were a marked bush (all 40 pictures) | 47 | 44 |

- **Imagery resolution matters.** AP with the model is 0.24 on 30 cm imagery, 0.22 on 50 cm and 0.20 on 60 cm.
- **Rows on the 13 hard empty fields:** one false "presumption" (field 3: lines in mottled soil),
  one "weak", 11 clean. Two filters helped: spots strung along a boundary (field edge, gully, soil
  band) are now rejected, and a very rare row gets 4x more random trials.
- **Ramparts are not detected yet.** The linear detectors cover 0-22% of the marked rampart.
  One fortress is not enough to build or test a rampart detector.

**Fourth pack (GEO: 25 village sites with 2-5 years each, 6 fortresses, 10 hard empty fields; scale,
dates and "visible / doubtful / not visible" for every picture).** Again tested with a model that
had never seen it.

Single pictures, by how well the marked things show according to the notes:

| the marked things are... | pictures | AP before model → with model | found among the top 30 | found at all |
|---|---|---|---|---|
| visible | 34 | 0.24 → 0.27 | 64% | 87% |
| doubtful | 36 | 0.19 → 0.21 | 52% | 72% |
| not visible | 9 | 0.10 → 0.13 | 41% | 71% |

**Several years of one place together.** This is what clears up doubtful pictures. With 2+ pictures
the list is now ranked first by how many pictures show a spot, then by how strong it is in them:

| | AP |
|---|---|
| one picture at a time (24 places of this pack) | 0.23 |
| all years together, only the strongest picture counts | 0.29 |
| all years together, old rule (strongest + bonus per extra picture) | 0.34 |
| all years together, new rule (count of pictures first) | **0.35** |
| only the *doubtful* years together (11 places): one at a time → together | 0.25 → 0.31 |

Over all four packs (55 places with 2-6 years): 0.22 one picture at a time, 0.35 together.
`sitecheck evaluate` prints this as "years together".

Still weak:
- **Bushes:** about 3 of the top-30 spots per picture sit on vegetation the user outlined.
- **Hard empty fields:** all 10 score "low" (2-23 points), but 3 of them get a false row
  "presumption" (lines in mottled soil), and single pictures show 4-18 spots scored ≥ 0.5.
  Their scale was not given. With the 13 earlier ones: 4 false presumptions on 23 hard empty fields.
- **The model on its own** ranks no better than plain spot strength on this pack (0.24 vs 0.25);
  on the three older packs it is slightly better (0.22-0.27 vs 0.18-0.26). The gain comes from
  combining it with the hand-made detectors, and much more from combining years.

### Ramparts and banks: closed outlines (`--outlines`)

`sitecheck image ... --outlines` (interface: "also draw closed outlines") looks for a closed
outline: from many trial centres it looks out in 72 directions and finds the smooth closed path
that follows as much edge (brightness, colour, texture) as possible. With several pictures it
keeps the outlines that enclose the same ground in 2+ of them.

Tested on 6 marked fortresses (28 pictures) and 25 village sites without one:
- At 5 of the 6 fortresses more than half of the repeated outline lies on the marked rampart
  (93, 77, 65, 57, 55 and 21%). The outline is ragged and follows one edge of the bank.
- **It cannot tell a fortress from anything else.** The same kind of repeated outline appears at
  22 of the 25 village sites (groves, ponds, plots, patches of different grass). Coverage,
  edge strength, a second parallel outline, standing out from the rest of the picture: none
  separates the two groups (0.5-0.6, where 0.5 is a coin toss).
- So outlines are drawn only on request and add nothing to the score. A rampart is a relief
  feature: with elevation data (`sitecheck run` / `analyze` on lidar) banks and ditches are found
  directly. For most of Russia there is no free elevation data fine enough.

The model is now trained on all four packs: 67 areas, 236 pictures, 25 empty places; bushes and
animals as weighted look-alikes; "not visible" pictures left out.

### Test it on places you know; retrain it on your own

```
my_marked/
  foundation/1/   1.png  2.png  3.png  notes.png  1local.txt
  field/4/        clear.png  marks.png  winter.png ...
  fortress/       1.png ... knotes.png  info.txt   <- a marked picture straight in the folder = one area
  Hard_Empty/     1.png ... 13.png  info.txt       <- unmarked pictures straight in a folder = nothing ever stood here
  emptyfield/1/   1.png ... 10.png                 <- an area folder without a marked copy: the same
```
- **Clean picture**: the one the marks were drawn on is found automatically: same size, most similar.
- **Marked copy**: `mark`, `note`, `mix` or `quad` in the name. Several marked copies are fine
  (e.g. `marks.png` + `FieldMarks.png` drawn on a ploughed-year picture): the marks are merged.
- **Other years/seasons**: same size = already aligned; otherwise aligned automatically.
- **Notes** (any `.txt` in the folder), for example:
  ```
  for all height 406 m
  window 1749 px
  coord (54 19 44 N 50 19 38 E)
  (for (notes,1) Maxar 6.25.2013 )
  (for (2) Maxar 8.31.2016)
  (for (3) Airbus 7.6.2024)
  visible: 1,2
  doubtful: 3
  not visible:
  ```
  - `height`: the **eye altitude**, which gives the screen scale (altitude / 1660 m per pixel for a
    1920 px wide map view, 60° field of view). `window` is the width of the map view in pixels;
    a `window` line in the pack's GeneralINFO.txt holds for every area. One line per picture also
    works: `7) height 1.34 km`.
  - `coord`: the place (it fixes the sun's height for each date).
  - Provider and date (month.day.year) per picture. The imagery's own resolution is estimated
    from them unless you write it (`images 50 cm in each pixel`). It decides what can be seen at
    all (nothing under ~3 imagery pixels), but it is not the screen scale.
  - `visible / doubtful / not visible`: how well the marked things show in each picture.
    "Not visible" pictures are left out of training (the model must not learn to see what is not
    in the pixels), and results are reported per class.
- **Colours** (when more than one colour is used, and the marks are drawn on the same picture):

  | colour | meaning |
  |---|---|
  | red loop | house pit |
  | yellow outline | remains of a foundation (several boxes sharing walls = one per box) |
  | red cross | animal (a look-alike: not a site) |
  | black outline | bush / trees (a look-alike) |
  | blue outline | river, lake (a spot found there is false) |
  | orange double line, zigzag between | road |
  | green line(s) | rampart / defensive bank (one line, or the inner and outer edge) |
  | white loop | "maybe, not sure": counts neither right nor wrong |
  | purple outline | the part of the picture that was checked; everything outside is ignored |

  One-colour (red) packs work as before: loops are objects, two lines are a road, a loop
  around 2+ marks is a yard, a zigzag is an area. Spots on a road or inside a yard count
  neither right nor wrong. Spots on a marked bush or animal count as false and are reported as
  "fooled by". In training they weigh 3x. `-o` writes a `*_marks.jpg` per area showing how
  your marks were read.

```bash
sitecheck evaluate my_marked -o eval_out              # per picture, per category, by visibility, years together
sitecheck train Rus_earth pack my_marked              # validates (area by area, and pack by pack), then saves
```
`train` needs `pip install scikit-learn` and prints the held-out scores first. Places closer than
3 km to each other (the same site shot twice, neighbouring sites) are always held out together, so
the test never trains on the place it is tested on. With 2+ packs it also prints the pack-by-pack
test and compares with plain spot strength. Keep the old model if
the new one isn't better (use `-o` to write elsewhere). `evaluate` on a pack the model was
trained on looks better than it really is.

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

## Fifth pack (Fear) and the image network (0.12)

The Fear pack: 74 new places (500 x 500 px, eye altitude 400 m, real scale) plus 25 earlier ones
again (these are recognised as the same pictures and counted once).

- **Tree model on a pack it has never seen.** Trained on the four older packs and tested on the 199
  new Fear pictures, it ranks spots WORSE than plain spot strength (AP 0.36 vs 0.47). Once pictures
  like these are in training, it is better (5-fold by place over all packs: 0.38 vs 0.34). So it
  learns the look of the packs it has seen and does not carry over well to another zoom, region or
  screenshot style - a reason to try the image network below, and to keep testing on new places.
- **Held-out test set** (33 places, 109 pictures, never used for training; list in
  `sitecheck/cnn_split.json`): current program 0.31 (hand-made detectors alone 0.29); by visibility
  0.44 visible, 0.32 doubtful, 0.28 not visible. All years of a place together: 0.40 vs 0.33 one
  picture at a time (30 places).

### Image network (`tools/train_cnn.ipynb`)

A small image network (ResNet-18 start, trained to put a peak on every pit and foundation) learns
straight from the pictures instead of from ~40 hand-made measurements. It cannot be trained in this
program (no GPU, no PyTorch) - it is trained once in a free GPU notebook and then runs inside
sitecheck with plain numpy (0.6 s per 500 x 500 picture, no internet, no AI service).

1. Kaggle (free, ~30 GPU hours a week): create a private Dataset with `sitecheck.zip` and your packs
   (Rus_earth, pack, Geo_Pack, GEO, Fear). New Notebook → File → Import `tools/train_cnn.ipynb` → add
   your dataset → Settings: Accelerator GPU, Internet on → Run all (about 1-2 hours).
   Colab or your own PC with an NVIDIA card work too (set INPUT / WORK at the top).
2. The notebook prepares the material (duplicates dropped, the same held-out places as above), trains
   40 epochs with rotations, flips, zoom 0.7-1.4 and colour changes, then scores the network on the
   held-out places against the current program's stored results (`sitecheck/cnn_baseline.json`).
3. Download `model_cnn.npz` from the notebook's output into `sitecheck/sitecheck/`. sitecheck uses it
   **only if** the network combined with the current program beat the program alone by more than 0.01
   on the held-out places; otherwise the file says `mode: off` and nothing changes. The notebook also
   checks that the numpy runner gives the same maps as PyTorch, and refuses to save otherwise.
   `report.json → per_image → image_network` shows whether it was used and its test scores.

Without `model_cnn.npz` everything works as before. `sitecheck cnn-prep packs... -o cnn_data`
prepares the material on your own computer if you prefer.

## Changes in 0.11 (code review)

Measured on the 82 marked pictures of the GEO pack (hand-made detectors, ranking quality):
- **Small shift between imagery dates corrected**: same-view pictures from different satellite
  passes sit a few metres apart (52 of 120 corrected, median 3.5 m). 0.20 → 0.23.
- **Scale with the Google Earth side panel open**: `window` line / `--view-width` (see above).
- **Sun from the picture date** is computed from the notes and reported, but NOT used for the
  shading relief: on these pictures it did not help (0.20 → 0.19) even though the sun estimated
  from shadows is off by ~50° on median. The relief from shading adds little; brightness and
  colour carry the detection.
- **A row of houses seen in one picture only** now gives at most 6 points (was up to 15). On the
  hard empty fields such single-picture rows were the main false alarm.
- **Same row counted twice** when found in two pictures: fixed. **Big soil patches** now merge
  across years (the merge distance grows with the patch size).
- **Training test** holds places within 3 km out together (foundation/14 and /15 have the same
  coordinates). Result unchanged (0.26), so the earlier numbers were not inflated.
- Interface showed "both pictures" for any repeat; now "in N of M pictures". The AI prompt
  explains what `confidence` means with several pictures.

## Tests

```bash
python tests/test_offline.py      # every downloader against a mock server + no-lidar tools
python tests/test_image_mode.py   # screenshot mode on a synthetic 'photo' with known pits (model off)
sitecheck evaluate Rus_earth      # real marked screenshots (see above)
```

Results:
- 28/28 checks pass, including the numpy tree model, the numpy image-network runner, a
  two-rows-and-a-street case, reading hand-drawn marks (one-colour and the full colour scheme), the
  notes file and a ring-shaped outline.
- Clean 1 m terrain: 30/30 pits, 9/10 foundations, 10/10 roads (10 runs).
- Point cloud with 15 % vegetation returns: 3/3 pits and the foundation.
- Surface model with forest: 246 candidates without the filter, 16 with it (pits found: 0 → 3).

**Not yet tested against the live services** (the build machine has no internet): 3DEP, the
Environment Agency, NLS, ArcticDEM STAC and Overpass. They follow the published APIs, and
the NLS and ArcticDEM clients discover process and collection names from the service itself.
The first real run will show any mismatch, with a clear error message.
