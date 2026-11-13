"""Offline tests: every downloader runs against a fake server that answers like the real
service (same URLs, parameters and response shapes), plus the new no-lidar tools.

    python tests/test_offline.py
"""
import io
import json
import math
import os
import sys
import tempfile
import zipfile
from urllib.parse import parse_qs, urlparse

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import make_synthetic as syn  # noqa: E402
from sitecheck import proj  # noqa: E402
from sitecheck.raster import load_dem, save_geotiff  # noqa: E402

TMP = tempfile.mkdtemp(prefix="sitecheck_test_")
WORLD = syn.build(1)          # 500 x 500 m terrain with known objects, centred on any spot
SPOT_PX = (220, 200)          # row, col of the spot inside WORLD


def terrain_at(east_m, north_m):
    """Elevation of the synthetic world at metres east/north of the spot (bilinear)."""
    from scipy.ndimage import map_coordinates
    rows = SPOT_PX[0] - np.asarray(north_m)
    cols = SPOT_PX[1] + np.asarray(east_m)
    return map_coordinates(WORLD, [rows, cols], order=1, mode="nearest")


def geotiff_bytes(z, x0, y0, dx, dy, epsg):
    p = os.path.join(TMP, "resp.tif")
    save_geotiff(z, x0, y0, dx, dy, p, epsg)
    with open(p, "rb") as fh:
        return fh.read()


class Resp:
    def __init__(self, status=200, content=b"", js=None, headers=None):
        self.status_code, self.content = status, content if js is None else json.dumps(js).encode()
        self.headers = headers or {"Content-Type": "application/json" if js is not None else "image/tiff"}
        self.text = self.content[:300].decode("latin1")

    def json(self):
        return json.loads(self.content)

    def iter_content(self, n):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeInternet:
    def __init__(self, spot, coarse_3dep=False, ea_years_missing=(2025, 2024, 2023)):
        self.lat, self.lon = spot
        self.coarse = coarse_3dep
        self.ea_missing = ea_years_missing
        self.log = []
        self.jobs = {}

    def _export(self, params):
        w, s, e, n = map(float, params["bbox"].split(","))
        W, H = map(int, params["size"].split(","))
        assert params.get("adjustAspectRatio") == "false", "aspect ratio must not be adjusted"
        lons = w + (np.arange(W) + .5) * (e - w) / W
        lats = n - (np.arange(H) + .5) * (n - s) / H
        LON, LAT = np.meshgrid(lons, lats)
        east = (LON - self.lon) * 111_320 * math.cos(math.radians(self.lat))
        north = (LAT - self.lat) * 110_574
        z = terrain_at(east, north).astype("float32")
        if self.coarse:   # 10 m source resampled to 1 m, like 3DEP where no lidar was flown
            from scipy.ndimage import zoom
            z = zoom(z[::10, ::10], 10, order=1)[:H, :W].astype("float32")
        return Resp(content=geotiff_bytes(z, w, n, (e - w) / W, (n - s) / H, 4326))

    def get(self, url, params=None, **kw):
        self.log.append(("GET", url.split("?")[0]))
        params = params or {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        if "nationalmap.gov" in url and url.endswith("/exportImage"):
            return self._export(params)
        if "nationalmap.gov" in url and url.endswith("/identify"):
            return Resp(js={"catalogItems": {"features": [{"attributes": {"ProductName": "1 meter DEM", "LowPS": 1}}]}})
        if "environment.data.gov.uk" in url:
            if any(f"_{y}_" in url for y in self.ea_missing):
                return Resp(status=404, content=b"<html>not found</html>", headers={})
            return self._export(params)
        if "maanmittauslaitos.fi" in url:
            return self._nls_get(url, {**kw, "params": params})
        return Resp(status=404, content=b"not found", headers={})

    def post(self, url, data=None, json=None, params=None, **kw):
        kw["params"] = params or {}
        self.log.append(("POST", url))
        if "overpass" in url:
            pts = [[self.lon + dx / (111_320 * math.cos(math.radians(self.lat))), self.lat - 260 / 110_574]
                   for dx in range(-300, 301, 50)]
            return Resp(js={"elements": [{"type": "way", "tags": {"waterway": "river", "name": "Test"},
                                          "geometry": [{"lon": a, "lat": b} for a, b in pts]}]})
        if "maanmittauslaitos.fi" in url:
            return self._nls_post(url, json, kw)
        return Resp(status=404, content=b"", headers={})

    # ---- NLS OGC API Processes ------------------------------------------------------
    def _check_key(self, kw):
        assert kw.get("params", {}).get("api-key") == "TESTKEY", "API key not sent"

    def _nls_get(self, url, kw):
        self._check_key(kw)
        base = "https://avoin-paikkatieto.maanmittauslaitos.fi/tiedostopalvelu/ogcproc/v1"
        if url.endswith("/processes"):
            return Resp(js={"processes": [{"id": "maastotietokanta_bbox"},
                                          {"id": "korkeusmalli_2m_bbox"}, {"id": "korkeusmalli_10m_bbox"}]})
        if url.endswith("/processes/korkeusmalli_2m_bbox"):
            return Resp(js={"id": "korkeusmalli_2m_bbox", "inputs": {
                "boundingBoxInput": {"schema": {"type": "object"}},
                "fileFormatInput": {"schema": {"type": "string", "enum": ["TIFF", "ASCII"]}}}})
        if "/jobs/" in url and url.endswith("/results"):
            return Resp(js={"results": [{"href": base + "/files/tiles.zip"}]})
        if "/jobs/" in url:
            return Resp(js={"status": "successful"})
        if url.endswith("/files/tiles.zip"):
            return Resp(content=self.jobs["zip"], headers={"Content-Type": "application/zip"})
        return Resp(status=404, content=b"", headers={})

    def _nls_post(self, url, body, kw):
        self._check_key(kw)
        inp = body["inputs"]
        assert inp["fileFormatInput"] == "TIFF"
        e0, n0, e1, n1 = inp["boundingBoxInput"]["bbox"]
        assert inp["boundingBoxInput"]["crs"].endswith("/3067")
        # two 2 m tiles in EPSG:3067 covering the bbox (split west/east)
        E0, N1 = math.floor(e0 / 2) * 2, math.ceil(n1 / 2) * 2
        W, H = int((e1 - E0) / 2) + 2, int((N1 - n0) / 2) + 2
        cols = E0 + (np.arange(W) + .5) * 2
        rows = N1 - (np.arange(H) + .5) * 2
        CE, CN = np.meshgrid(cols, rows)
        lon, lat = proj.inverse(3067, CE, CN)
        east = (lon - self.lon) * 111_320 * math.cos(math.radians(self.lat))
        north = (lat - self.lat) * 110_574
        z = terrain_at(east, north).astype("float32")
        half = W // 2
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("a/tile_w.tif", geotiff_bytes(z[:, :half], E0, N1, 2, 2, 3067))
            zf.writestr("a/tile_e.tif", geotiff_bytes(z[:, half:], E0 + half * 2, N1, 2, 2, 3067))
        self.jobs["zip"] = buf.getvalue()
        return Resp(status=201, js={"jobID": "j1", "status": "accepted"},
                    headers={"Content-Type": "application/json",
                             "Location": url.rsplit("/processes/", 1)[0] + "/jobs/j1"})


def install(fake):
    requests.get, requests.post = fake.get, fake.post


def objects_found(report):
    F = report["features"]
    # synthetic objects relative to spot (m east, m north)
    pits = [(150 - 200, 220 - 200), (175 - 200, 220 - 215), (320 - 200, 220 - 180)]
    hit = lambda kind, x, y, tol=5: any(f["type"] == kind and math.hypot(f["x_m"] - x, f["y_m"] - y) < tol for f in F)
    return (sum(hit("pit", x, y) for x, y in pits), hit("rectangle", 240 - 200, 220 - 250),
            any(f["type"] == "linear" and f.get("straightness", 0) > .9 and math.hypot(f["x_m"] + 50, f["y_m"] - 115) < 15 for f in F))


def run_cli(args):
    from sitecheck.cli import main
    import contextlib
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = main(args)
    return rc, out.getvalue()


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    return bool(cond)


def main():
    import warnings
    warnings.filterwarnings("ignore")
    ok = True

    # ---- USA: routing, 3DEP, water ------------------------------------------------------
    spot = (44.9271, -93.2650)
    fake = FakeInternet(spot); install(fake)
    out = os.path.join(TMP, "us")
    rc, txt = run_cli(["run", "--point", f"{spot[0]},{spot[1]}", "--radius", "250", "-o", out])
    rep = json.load(open(os.path.join(out, "report.json")))
    p, r, road = objects_found(rep)
    ok &= check("USA: 3DEP chosen automatically", rep["data"]["download"]["dem"]["source"] == "3dep")
    ok &= check("USA: true 1 m pixels (aspect fix)", abs(rep["data"]["dem"]["cell_size_m"] - 1.0) < 0.05,
                rep["data"]["dem"]["cell_size_m"])
    ok &= check("USA: objects found through the download path", p == 3 and r and road, f"pits {p}/3 rect {r} road {road}")
    ok &= check("USA: water from OSM", rep["water"]["distance_m"] and 250 < rep["water"]["distance_m"] < 280,
                rep["water"]["distance_m"])

    fake = FakeInternet(spot, coarse_3dep=True); install(fake)
    out = os.path.join(TMP, "us_coarse")
    rc, txt = run_cli(["run", "--point", f"{spot[0]},{spot[1]}", "--radius", "250", "-o", out])
    rep = json.load(open(os.path.join(out, "report.json")))
    ok &= check("USA: area without lidar is flagged as resampled",
                any("resampled" in w for w in rep["warnings"]), rep["data"]["dem"]["interpolation_check"])

    # ---- England: EA with fallback over service names -------------------------------------
    spot = (51.75, -1.26)
    fake = FakeInternet(spot); install(fake)
    out = os.path.join(TMP, "uk")
    rc, txt = run_cli(["run", "--point", f"{spot[0]},{spot[1]}", "--radius", "250", "-o", out])
    rep = json.load(open(os.path.join(out, "report.json")))
    p, r, road = objects_found(rep)
    ok &= check("England: EA chosen, falls back to the 2022 composite",
                rep["data"]["download"]["dem"]["source"] == "ea" and "2022" in rep["data"]["dem"]["label"],
                rep["data"]["dem"]["label"])
    ok &= check("England: objects found", p == 3 and r and road, f"pits {p}/3 rect {r} road {road}")

    # ---- Finland: NLS async job, zip of 2 tiles in EPSG:3067, built-in projection ---------
    spot = (60.45, 25.10)
    fake = FakeInternet(spot); install(fake)
    out = os.path.join(TMP, "fi")
    rc, txt = run_cli(["run", "--point", f"{spot[0]},{spot[1]}", "--radius", "250", "-o", out, "--nls-key", "TESTKEY"])
    rep = json.load(open(os.path.join(out, "report.json")))
    p, r, road = objects_found(rep)
    ok &= check("Finland: NLS job ran, tiles joined", rep["data"]["download"]["dem"]["source"] == "nls"
                and rep["data"]["dem"]["epsg"] == 3067, rep["data"]["dem"]["label"])
    ok &= check("Finland: 2 m cells", abs(rep["data"]["dem"]["cell_size_m"] - 2) < 0.01)
    ok &= check("Finland: pits and road visible at 2 m", p >= 2 and road, f"pits {p}/3 rect {r} road {road}")

    fake = FakeInternet(spot); install(fake)
    from sitecheck.fetch import plan
    ok &= check("Finland without key: skips NLS", "nls" not in plan(60.45, 25.10, False))
    ok &= check("Russia (Karelia): ArcticDEM first", plan(62.0, 34.0)[0] == "arcticdem")
    ok &= check("Russia (Moscow region): GLO-30", plan(55.7, 37.6) == ["glo30"])

    # ---- Russia toolkit 1: drone/phone point cloud with trees -> DEM -------------------------
    rng = np.random.default_rng(3)
    n = 400_000
    east, north = rng.uniform(-200, 200, n), rng.uniform(-150, 150, n)
    z = terrain_at(east, north)
    canopy = rng.random(n) < 0.15                      # 15 % of returns hit vegetation
    z = z + np.where(canopy, rng.uniform(2, 15, n), 0) + rng.normal(0, 0.03, n)
    xyz = os.path.join(TMP, "scan.xyz")
    np.savetxt(xyz, np.column_stack([east, north, z]), fmt="%.3f")
    dem_path = os.path.join(TMP, "scan_dem.tif")
    spot = (56.10, 40.40)  # Vladimir oblast - no lidar, no ArcticDEM
    rc, txt = run_cli(["grid", xyz, "--anchor", f"{spot[0]},{spot[1]}", "--cell", "0.75", "-o", dem_path])
    out = os.path.join(TMP, "ru_scan")
    rc, txt = run_cli(["analyze", "--point", f"{spot[0]},{spot[1]}", "--dem", dem_path, "--radius", "140",
                       "-o", out, "--dem-label", "phone/drone scan"])
    rep = json.load(open(os.path.join(out, "report.json")))
    p, r, road = objects_found(rep)
    ok &= check("Point cloud (XYZ, local coords, 15 % vegetation hits): pits + foundation found",
                p >= 2 and r, f"pits {p}/3 rect {r}")

    # binary PLY reader
    ply = os.path.join(TMP, "scan.ply")
    m = 5000
    with open(ply, "wb") as fh:
        fh.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {m}\nproperty float x\n"
                 f"property float y\nproperty float z\nproperty uchar red\nend_header\n".encode())
        rec = np.zeros(m, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1")])
        rec["x"], rec["y"], rec["z"] = east[:m], north[:m], z[:m]
        fh.write(rec.tobytes())
    from sitecheck.pointcloud import read_points
    pts, _ = read_points(ply)
    ok &= check("PLY binary reader", pts.shape == (m, 3) and np.allclose(pts[:, 2], z[:m], atol=1e-3))

    # ---- Russia toolkit 2: surface model with forest -> ground filter -------------------------
    yy, xx = np.mgrid[0:500, 0:500]
    t = np.zeros((500, 500))
    for _ in range(900):
        cx, cy = rng.uniform(0, 500), rng.uniform(0, 300)
        h, rr = rng.uniform(4, 18), rng.uniform(1.5, 4)
        t = np.maximum(t, h * np.clip(1 - ((xx - cx) ** 2 + (yy - cy) ** 2) / rr ** 2, 0, 1) ** .5)
    dsm_path = os.path.join(TMP, "dsm.tif")
    syn.save_geotiff((WORLD + t).astype("float32"), dsm_path)
    lon, lat = syn.spot_lonlat(200, 220)
    res = {}
    for gf in ("off", "auto"):
        out = os.path.join(TMP, f"dsm_{gf}")
        run_cli(["analyze", "--point", f"{lat},{lon}", "--dem", dsm_path, "--dsm", "--ground-filter", gf, "-o", out])
        res[gf] = json.load(open(os.path.join(out, "report.json")))
    n_off, n_on = len(res["off"]["features"]), len(res["auto"]["features"])
    p_off, _, _ = objects_found(res["off"])
    p_on, _, _ = objects_found(res["auto"])
    ok &= check("Ground filter: far fewer false candidates and pits recovered",
                n_on < n_off / 3 and p_on > p_off, f"candidates {n_off} -> {n_on}, pits {p_off} -> {p_on}")

    # ---- Old map overlay (in a different CRS than the DEM) ------------------------------------
    E, N = proj.forward(3067, lon, lat)
    img = np.full((300, 300, 3), 235, np.uint8)
    # draw the 'farmstead' symbol where the foundation is (40 m E, 30 m S of spot), 1 px = 2 m
    r0, c0 = 150 + 15, 150 + 20
    img[r0 - 4:r0 + 4, c0 - 6:c0 + 6] = (180, 40, 40)
    om = os.path.join(TMP, "old_map.tif")
    from PIL import Image
    from PIL.TiffImagePlugin import ImageFileDirectory_v2
    ifd = ImageFileDirectory_v2()
    ifd[33550] = (2.0, 2.0, 0.0)
    ifd[33922] = (0.0, 0.0, 0.0, float(E - 300), float(N + 300), 0.0)
    ifd[34735] = (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 3067)
    ifd.tagtype[34735] = 3
    Image.fromarray(img).save(om, tiffinfo=ifd)
    out = os.path.join(TMP, "overlay")
    run_cli(["analyze", "--point", f"{lat},{lon}", "--dem", os.path.join(HERE, "..", "examples", "synthetic_dem.tif"),
             "--overlay", om, "-o", out])
    lay = np.load(os.path.join(out, "layers.npz"))
    ov = lay["overlay"]
    rr, cc = np.argwhere((ov[..., 0] == 180) & (ov[..., 3] == 255)).mean(axis=0)
    rc_spot = lay["spot_rc"]
    dx, dy = cc - rc_spot[1], rc_spot[0] - rr
    ok &= check("Old map (EPSG:3067) lands in the right place on the DEM (EPSG:4326)",
                abs(dx - 40) < 3 and abs(dy + 30) < 3, f"symbol at {dx:.1f} m E, {dy:.1f} m N; expected 40, -30")
    ok &= check("overlay.png written", os.path.exists(os.path.join(out, "overlay.png")))

    # ---- Learned spot model (numpy-only tree evaluation) --------------------------------------
    from sitecheck import learned as LM
    mdl = LM.load_model()
    ok &= check("Learned spot model ships with the package", mdl is not None and mdl.get("type") == "gbt",
                f"{len(mdl['trees']) if mdl else 0} trees, validation {mdl.get('validation') if mdl else None}")
    rng = np.random.default_rng(5)
    field = np.clip(0.42 + 0.03 * rng.standard_normal((300, 400, 3)), 0, 1).astype(np.float32)
    field[..., 1] *= 0.9
    yy, xx = np.mgrid[:300, :400]
    for c_ in range(60, 360, 60):                               # light patches
        field[(yy - 100) ** 2 + (xx - c_) ** 2 < 12 ** 2] += 0.12
    spots = LM.detect(np.clip(field, 0, 1), model=mdl, min_p=0.0, max_n=200)
    ok &= check("Learned detector runs without scikit-learn and scores spots in [0,1]",
                len(spots) > 0 and all(0 <= s_["confidence"] <= 1 for s_ in spots), f"{len(spots)} spots")

    # ---- Two parallel rows of patches = a street --------------------------------------------
    from sitecheck import patterns as PT
    mpp_ = 0.5
    img = np.clip(0.40 + 0.025 * rng.standard_normal((500, 700, 3)), 0, 1).astype(np.float32)
    img[..., 0] *= 1.15
    gy, gx = np.mgrid[:500, :700]
    for row_r in (150, 330):                                    # 90 m apart, houses every 50 m
        for c_ in range(80, 640, 100):
            img[(gy - row_r) ** 2 + (gx - c_) ** 2 < 14 ** 2] += 0.15
    img = np.clip(img, 0, 1)
    region = np.ones(img.shape[:2], bool)
    bl, maps = PT.find_blobs(img, region, mpp_, max_m=35.0, return_maps=True)
    rows = PT.find_rows(bl, region, mpp_, maps=maps)
    best = rows[0] if rows else None
    par = PT.parallel_row(best, maps, region, mpp_) if best else None
    ok &= check("Row of patches found, and the parallel row on the other side of the street",
                best is not None and par is not None and par["chance"] < 0.05 and abs(par["offset_m"] - 90) < 12,
                f"row chance {best['chance'] if best else None}, parallel offset "
                f"{par['offset_m'] if par else None} m, chance {par['chance'] if par else None}")

    # ---- Reading the user's red marks: houses, a road (two lines), a yard, an empty place ----
    import cv2
    from sitecheck import groundtruth as GT
    base = np.clip(0.45 + 0.02 * rng.standard_normal((400, 800, 3)), 0, 1).astype(np.float32)
    marked = (base * 255).astype(np.uint8).copy()
    red = (255, 0, 0)
    cv2.polylines(marked, [np.array([[10, 200], [400, 190], [790, 205]], np.int32)], False, red, 3)   # road
    cv2.polylines(marked, [np.array([[10, 230], [400, 220], [790, 235]], np.int32)], False, red, 3)
    cv2.circle(marked, (300, 207), 9, red, 2)                       # a pit circled right on the road
    for x in (100, 250, 400, 550):                                   # houses above the road
        cv2.rectangle(marked, (x, 120), (x + 50, 160), red, 3)
    cv2.ellipse(marked, (200, 320), (120, 50), 0, 0, 360, red, 3)    # a yard with 2 pits inside
    cv2.circle(marked, (160, 320), 12, red, 2)
    cv2.circle(marked, (240, 320), 12, red, 2)
    cv2.ellipse(marked, (650, 320), (40, 25), 0, 20, 330, red, 3)    # an oval left unclosed
    g = GT.extract(base, marked.astype(np.float32) / 255)
    rects = sum(o["shape"] == "rectangle" for o in g["objects"])
    ok &= check("Red marks: 4 houses + 3 pits + unclosed oval, road from two lines, yard not an object",
                len(g["objects"]) == 8 and rects == 4 and len(g["streets"]) == 1 and g["streets"][0]["double_line"]
                and g["enclosures"].any() and g["roads"][215, 600],
                f"{len(g['objects'])} objects ({rects} rectangles), {len(g['streets'])} road(s), "
                f"yard {'yes' if g['enclosures'].any() else 'no'}")

    # ---- Colour marking scheme + scale notes ------------------------------------------------
    from sitecheck import scale as SCL
    notes = SCL.parse_notes("for all height 473 m\n(for (notes,1) images 30 cm in each pixel )\n"
                            "(for (2,3) images 50 cm in each pixel )", ["1.png", "2.png", "3.png", "notes.png"])
    per = SCL.parse_notes("(for All images 30 cm in each pixel )\n1) height 493 m\n7) height 1.34 km",
                          ["1.png", "7.png"])
    ok &= check("Scale notes: eye altitude -> m/px, imagery resolution per picture",
                abs(notes["2.png"]["mpp"] - 0.2845) < 0.002 and notes["2.png"]["imagery_m"] == 0.5
                and notes["notes.png"]["imagery_m"] == 0.3 and abs(per["7.png"]["altitude_m"] - 1340) < 1,
                f"473 m -> {notes['2.png']['mpp']} m/px; 1.34 km -> {per['7.png']['mpp']} m/px")
    base2 = np.clip(0.45 + 0.02 * rng.standard_normal((400, 800, 3)), 0, 1).astype(np.float32)
    mk = (base2 * 255).astype(np.uint8).copy()
    YEL, RED, BLK, ORA, GRN = (255, 240, 0), (255, 0, 0), (0, 0, 0), (255, 140, 0), (0, 230, 0)
    for x in (100, 200, 300):
        cv2.rectangle(mk, (x, 60), (x + 40, 95), YEL, 2)              # 3 foundations
    for x in (450, 520):
        cv2.circle(mk, (x, 80), 12, RED, 2)                           # 2 pits
    cv2.ellipse(mk, (650, 80), (40, 25), 0, 0, 360, BLK, 2)           # a bush
    cv2.line(mk, (700, 150), (720, 170), RED, 2)
    cv2.line(mk, (720, 150), (700, 170), RED, 2)                      # an animal (X)
    cv2.polylines(mk, [np.array([[10, 200], [790, 210]], np.int32)], False, ORA, 2)    # road
    cv2.polylines(mk, [np.array([[10, 230], [790, 240]], np.int32)], False, ORA, 2)
    zz = np.array([[20 + 30 * i, 203 if i % 2 else 233] for i in range(25)], np.int32)
    cv2.polylines(mk, [zz], False, ORA, 2)                            # zigzag inside the road
    cv2.polylines(mk, [np.array([[10, 300], [790, 300]], np.int32)], False, GRN, 2)    # rampart
    cv2.polylines(mk, [np.array([[10, 340], [790, 340]], np.int32)], False, GRN, 2)
    g2 = GT.extract(base2, mk.astype(np.float32) / 255)
    kinds = [o["kind"] for o in g2["objects"]]
    la = sorted(x["kind"] for x in g2["lookalikes"])
    ok &= check("Colour marks: yellow=foundation, red=pit, black=bush, red X=animal, orange=road, green=rampart",
                kinds.count("foundation") == 3 and kinds.count("pit") == 2 and la == ["animal", "vegetation"]
                and g2["roads"][220, 400] and g2["ramparts"][320, 400],
                f"{kinds.count('foundation')} foundations, {kinds.count('pit')} pits, look-alikes {la}, "
                f"road {'yes' if g2['roads'][220, 400] else 'no'}, rampart {'yes' if g2['ramparts'][320, 400] else 'no'}")

    # ---- Notes: provider + date, coordinates, visibility --------------------------------------
    n3 = SCL.parse_notes("for all height 1.09 km\ncoord (53 14 01 N 38 33 26 E)\nsize for all screenshot 992 x 532\n"
                         "(for (notes,1) CNES/Airbus 5.31.2014 )\n(for (2 Maxar 10.8.2020)\n(for (3,4 Maxar 9.17.2006)\n"
                         "visible: 1\ndoubtful: 2 3\nnot visible: 4\n",
                         ["1.png", "2.png", "3.png", "4.png", "notes.png"])
    ok &= check("Notes: date and provider per picture, coordinates, visible / doubtful / not visible",
                n3["2.png"]["date"] == "2020-10-08" and n3["1.png"]["provider"] == "CNES/Airbus"
                and abs(n3["_area"]["lat"] - 53.2336) < 0.001 and abs(n3["_area"]["lon"] - 38.5572) < 0.001
                and n3["1.png"]["visibility"] == "visible" and n3["3.png"]["visibility"] == "doubtful"
                and n3["4.png"]["visibility"] == "not visible" and n3["4.png"]["date"] == "2006-09-17"
                and abs(n3["1.png"]["mpp"] - 0.6555) < 0.002,
                f"{n3['2.png']['date']} {n3['1.png']['provider']}, {n3['_area']['lat']} {n3['_area']['lon']}, "
                f"sun {n3['2.png'].get('sun_elevation')} deg")

    # ---- More colours: white = not sure, purple = checked area, blue = water --------------------
    base3 = np.clip(0.45 + 0.02 * rng.standard_normal((400, 800, 3)), 0, 1).astype(np.float32)
    mk3 = (base3 * 255).astype(np.uint8).copy()
    cv2.ellipse(mk3, (400, 200), (330, 160), 0, 0, 360, (200, 0, 255), 2)      # purple: what was checked
    cv2.circle(mk3, (300, 200), 12, (255, 0, 0), 2)                             # a pit
    cv2.ellipse(mk3, (450, 200), (30, 20), 0, 0, 360, (255, 255, 255), 2)       # white: maybe
    cv2.ellipse(mk3, (560, 220), (40, 30), 0, 0, 360, (0, 0, 255), 2)           # blue: pond
    g3 = GT.extract(base3, mk3.astype(np.float32) / 255)
    from sitecheck import evaluate as EVt
    cands3 = [{"row": 200, "col": 300, "confidence": .9}, {"row": 200, "col": 450, "confidence": .8},
              {"row": 20, "col": 20, "confidence": .7}, {"row": 100, "col": 400, "confidence": .6},
              {"row": 220, "col": 560, "confidence": .5}]
    sc3, _ = EVt.match(cands3, g3, (400, 800))
    ok &= check("White loop and outside the purple outline are ignored; a spot in marked water is false",
                len(g3["objects"]) == 1 and g3["unsure"][200, 450] and g3["checked"][200, 400]
                and not g3["checked"][20, 20] and g3["water"][220, 560]
                and [x[1] for x in sc3] == [True, False, False],
                f"{len(g3['objects'])} pit, scored spots: {[x[1] for x in sc3]} (of 5 given)")

    # ---- Closed outline (rampart-like ring) ---------------------------------------------------
    from sitecheck import enclosure as ENC
    ring = np.clip(0.42 + 0.015 * rng.standard_normal((500, 600, 3)), 0, 1).astype(np.float32)
    gy5, gx5 = np.mgrid[:500, :600]
    rad5 = np.hypot((gy5 - 250) / 0.8, gx5 - 300)
    ring[(rad5 > 150) & (rad5 < 170)] += 0.10                                   # a lighter band, 60 x 48 m radius
    outl = ENC.find_enclosures(np.clip(ring, 0, 1), np.ones((500, 600), bool), 0.4, max_n=1)
    o5 = outl[0] if outl else None
    ok &= check("Closed outline: a ring-shaped band is traced all the way round",
                o5 is not None and o5["coverage"] > 0.9 and abs(o5["row"] - 250) < 40 and abs(o5["col"] - 300) < 40
                and 80 < o5["size_m"][0] < 160,
                f"coverage {o5['coverage'] if o5 else None}, size {o5['size_m'] if o5 else None} m")

    # ---- Image network: numpy runner, training targets ------------------------------------------
    from sitecheck import cnn as CNN, cnn_data as CD
    xx = rng.random((3, 9, 8)).astype(np.float32)
    ww = rng.random((4, 3, 3, 3)).astype(np.float32)
    yy_ = CNN.conv2d(xx, ww, np.zeros(4, np.float32), 2, 1)
    xp_ = np.pad(xx, ((0, 0), (1, 1), (1, 1)))
    ref_ = np.array([[[(xp_[:, 2 * i:2 * i + 3, 2 * j:2 * j + 3] * ww[o]).sum() for j in range(yy_.shape[2])]
                      for i in range(yy_.shape[1])] for o in range(4)])
    rm = CNN.random_model()
    hm_ = CNN.heatmaps(rng.random((133, 171, 3)).astype(np.float32), rm, mpp=0.3)
    tg = CD.targets(np.array([[50, 60, 12, 0], [100, 30, 40, 1]], np.float32), (160, 160))
    ok &= check("Image network: numpy layers exact, any picture size, every object gets a target peak",
                np.abs(yy_ - ref_).max() < 1e-4 and hm_.shape == (2, 133, 171)
                and tg[0].max() == 1.0 and tg[1].max() == 1.0 and tg.shape == (2, 40, 40),
                f"conv diff {np.abs(yy_ - ref_).max():.1e}, heat maps {hm_.shape}")

    print("\nALL PASSED" if ok else "\nSOME TESTS FAILED")
    print("outputs in", TMP)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
