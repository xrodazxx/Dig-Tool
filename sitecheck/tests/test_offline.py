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

    print("\nALL PASSED" if ok else "\nSOME TESTS FAILED")
    print("outputs in", TMP)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
