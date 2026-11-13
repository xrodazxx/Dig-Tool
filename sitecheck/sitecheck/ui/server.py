"""Local web interface: `sitecheck ui --workspace DIR`. Runs on this computer only
(127.0.0.1), works offline, opens in the browser.

Workspace layout
    DIR/areas/<name>/input/    screenshots + settings.json
    DIR/areas/<name>/result/   analysis output (report.json, layers/, ...)
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import threading
import time
import traceback

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

from ..batch import IMG_EXT, _natural, _sharpness, attribution_mask, parse_coords

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _safe(name: str) -> str:
    name = re.sub(r"[^\w\-. ()]+", "_", name, flags=re.UNICODE).strip(" .")
    if not name or name in (".", ".."):
        abort(400, "bad name")
    return name


def create_app(workspace: str) -> Flask:
    app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
    areas_root = os.path.join(workspace, "areas")
    os.makedirs(areas_root, exist_ok=True)

    def adir(a):
        d = os.path.join(areas_root, _safe(a))
        if not os.path.isdir(d):
            abort(404, f"no area {a}")
        return d

    def settings_path(a):
        return os.path.join(adir(a), "input", "settings.json")

    def load_settings(a):
        p = settings_path(a)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def save_settings(a, s):
        with open(settings_path(a), "w", encoding="utf-8") as fh:
            json.dump(s, fh, ensure_ascii=False, indent=1)

    def images_of(a):
        s = load_settings(a)
        files = [f for f in os.listdir(os.path.join(adir(a), "input")) if f.lower().endswith(IMG_EXT)]
        order = [f for f in s.get("images", []) if f in files]
        return order + sorted([f for f in files if f not in order], key=_natural)

    def new_area(name, coords_text, files_from):
        name = _safe(name)
        base = os.path.join(areas_root, name)
        i = 2
        while os.path.exists(base):
            base = os.path.join(areas_root, f"{name}_{i}")
            i += 1
        inp = os.path.join(base, "input")
        os.makedirs(inp)
        os.makedirs(os.path.join(base, "result"))
        for f in files_from:
            if isinstance(f, str):
                shutil.copy(f, inp)
            else:
                f.save(os.path.join(inp, _safe(os.path.basename(f.filename))))
        imgs = sorted([os.path.join(inp, f) for f in os.listdir(inp) if f.lower().endswith(IMG_EXT)],
                      key=_sharpness, reverse=True)
        s = {"images": [os.path.basename(p) for p in imgs], "lat": None, "lon": None,
             "mpp": None, "known": None, "masks": [list(attribution_mask(imgs[0]))] if imgs else [],
             "spot_px": None, "sun_azimuth": None, "coords_text": coords_text or ""}
        if coords_text:
            try:
                s["lat"], s["lon"] = parse_coords(coords_text)
            except ValueError:
                pass
        with open(os.path.join(inp, "settings.json"), "w", encoding="utf-8") as fh:
            json.dump(s, fh, ensure_ascii=False, indent=1)
        return os.path.basename(base)

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    # ------------------------------------------------------------------ areas
    @app.get("/api/areas")
    def list_areas():
        out = []
        for d in sorted(glob.glob(os.path.join(areas_root, "*")), key=lambda x: _natural(os.path.basename(x))):
            if not os.path.isdir(d):
                continue
            a = os.path.basename(d)
            rp = os.path.join(d, "result", "report.json")
            item = {"name": a, "analysed": os.path.exists(rp), "status": JOBS.get(a, {}).get("state")}
            if item["analysed"]:
                try:
                    with open(rp, encoding="utf-8") as fh:
                        r = json.load(fh)
                    item["score"] = r["score"]["points"]
                    item["band"] = r["score"]["band"]
                    item["rows"] = sum(1 for f in r["features"] if f["type"] == "row")
                except Exception:  # noqa: BLE001
                    pass
            out.append(item)
        return jsonify(out)

    @app.post("/api/areas")
    def add_area():
        files = request.files.getlist("images")
        if not files:
            abort(400, "no images")
        name = new_area(request.form.get("name") or "area", request.form.get("coords", ""), files)
        return jsonify({"name": name})

    @app.post("/api/import")
    def import_folder():
        """Folder with one sub-folder per area (images + coordinates .txt), like `batch`."""
        path = (request.json or {}).get("path", "")
        if not os.path.isdir(path):
            abort(400, f"not a folder: {path}")
        subs = sorted([d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d)], key=_natural) or [path]
        made = []
        for d in subs:
            imgs = [f for f in glob.glob(os.path.join(d, "*")) if f.lower().endswith(IMG_EXT)]
            if not imgs:
                continue
            txt = ""
            for t in glob.glob(os.path.join(d, "*.txt")):
                with open(t, encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read()
                break
            made.append(new_area(os.path.basename(d.rstrip("/")), txt, imgs))
        return jsonify({"created": made})

    @app.get("/api/areas/<a>")
    def get_area(a):
        d = adir(a)
        res = os.path.join(d, "result")
        out = {"name": a, "settings": load_settings(a), "images": images_of(a),
               "job": JOBS.get(a), "report": None, "features": None, "layers": []}
        rp = os.path.join(res, "report.json")
        if os.path.exists(rp):
            with open(rp, encoding="utf-8") as fh:
                out["report"] = json.load(fh)
            with open(os.path.join(res, "features_full.json"), encoding="utf-8") as fh:
                out["features"] = json.load(fh)
            out["layers"] = sorted(os.path.basename(p) for p in glob.glob(os.path.join(res, "layers", "*.jpg")))
        return jsonify(out)

    @app.post("/api/areas/<a>/settings")
    def set_settings(a):
        s = load_settings(a)
        body = request.json or {}
        for k in ("lat", "lon", "mpp", "known", "masks", "spot_px", "sun_azimuth", "images", "coords_text", "outlines"):
            if k in body:
                s[k] = body[k]
        if body.get("coords_text"):
            try:
                s["lat"], s["lon"] = parse_coords(body["coords_text"])
            except ValueError as ex:
                abort(400, str(ex))
        if body.get("altitude"):                     # Google Earth eye altitude -> m/px
            from ..scale import mpp_from_altitude
            s["altitude"] = float(body["altitude"])
            s["mpp"] = round(mpp_from_altitude(s["altitude"], int(body.get("view_width") or 1920)), 4)
            s["known"] = None
        if s.get("known"):
            x1, y1, x2, y2, m = s["known"]
            s["mpp"] = round(float(m) / max(1e-6, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5), 4)
        save_settings(a, s)
        return jsonify(s)

    @app.delete("/api/areas/<a>")
    def delete_area(a):
        shutil.rmtree(adir(a))
        return jsonify({"deleted": a})

    # ------------------------------------------------------------------ analysis
    def _run(a):
        from ..image_mode import run_images
        try:
            s = load_settings(a)
            d = adir(a)
            paths = [os.path.join(d, "input", f) for f in images_of(a)]
            res = os.path.join(d, "result")
            shutil.rmtree(res, ignore_errors=True)
            JOBS[a].update(state="running", step="analysing screenshots")
            run_images(paths, res, s.get("lat"), s.get("lon"), mpp=s.get("mpp"),
                       spot_px=tuple(s["spot_px"]) if s.get("spot_px") else None,
                       masks=[tuple(m) for m in s.get("masks") or []], sun_az=s.get("sun_azimuth"),
                       outlines=bool(s.get("outlines")))
            JOBS[a].update(state="done", finished=time.time())
        except Exception as ex:  # noqa: BLE001
            JOBS[a].update(state="error", error=str(ex), trace=traceback.format_exc()[-2000:])

    @app.post("/api/areas/<a>/run")
    def run(a):
        adir(a)
        with _LOCK:
            if JOBS.get(a, {}).get("state") == "running":
                return jsonify(JOBS[a])
            JOBS[a] = {"state": "running", "started": time.time()}
        threading.Thread(target=_run, args=(a,), daemon=True).start()
        return jsonify(JOBS[a])

    @app.get("/api/areas/<a>/job")
    def job(a):
        return jsonify(JOBS.get(a) or {"state": "idle"})

    # ------------------------------------------------------------------ review
    @app.post("/api/areas/<a>/review")
    def do_review(a):
        from ..review import review
        res = os.path.join(adir(a), "result")
        if not os.path.exists(os.path.join(res, "report.json")):
            abort(400, "run the analysis first")
        b = request.json or {}
        adds = []
        if b.get("add"):
            lay = np.load(os.path.join(res, "layers.npz"))
            rc, cell = lay["spot_rc"], float(lay["cell"])
            for item in b["add"]:
                x = (item["col"] - rc[1]) * cell
                y = (rc[0] - item["row"]) * cell
                adds.append(f"{item.get('type', 'pit')}:{x:.1f},{y:.1f}:{item.get('size', 5)}")
        rep = review(res, b.get("confirm", []), b.get("reject", []), adds,
                     extra_features=b.get("extra_features", []), unreview=b.get("unreview", []))
        return jsonify({"score": rep["score"]})

    @app.post("/api/areas/<a>/guided")
    def guided(a):
        from .. import imagery as im
        from ..patterns import guided_row
        res = os.path.join(adir(a), "result")
        if not os.path.exists(os.path.join(res, "report.json")):
            abort(400, "run the analysis first")
        b = request.json or {}
        seeds = b.get("seeds") or []
        if len(seeds) < 2:
            abort(400, "click at least 2 patches")
        s = load_settings(a)
        with open(os.path.join(res, "report.json"), encoding="utf-8") as fh:
            rep = json.load(fh)
        mpp = rep["data"]["metres_per_pixel"]
        rgb, valid = im.load(os.path.join(res, "layers", "picture1.jpg"),
                             [tuple(m) for m in s.get("masks") or []])
        reg = np.load(os.path.join(res, "layers", "region.npy"))
        lay = np.load(os.path.join(res, "layers.npz"))
        relief = lay["z"]
        v = relief[reg & (relief != 0)]
        extra = {}
        if v.size > 100:
            mad = 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-6
            extra["hollow"] = np.where(reg, -relief / mad, 0).astype("float32")
        fam = b.get("family") if b.get("family") not in (None, "", "auto") else None
        try:
            row = guided_row(rgb, reg & valid, mpp, seeds, extra_maps=extra, family=fam)
        except ValueError as ex:
            abort(400, str(ex))
        row["n_images"] = len(rep["data"]["images"])
        row["seen_in_images"] = [1]
        # independent second look in the other aligned pictures
        from ..patterns import check_row_in_other_picture
        checks = []
        for p2 in sorted(glob.glob(os.path.join(res, "layers", "picture*.jpg")))[1:]:
            rgb2, valid2 = im.load(p2)
            cls2 = im.classify(rgb2, valid2, mpp)
            reg2 = (cls2["field"] | cls2["open"]) & ~cls2["snow"] & (rgb2.max(-1) > 0.02)
            ck = check_row_in_other_picture(rgb2, reg2, mpp, row)
            if ck:
                ck["picture"] = int(re.findall(r"\d+", os.path.basename(p2))[0])
                checks.append(ck)
        row["other_pictures"] = checks
        for ck in checks:
            if ck["hits"] >= 2 and ck["chance"] < 0.5:
                row["seen_in_images"].append(ck["picture"])
                row["chance"] = round(row["chance"] * ck["chance"], 4)
        if checks:
            ck = checks[0]
            row["note"] += (f" In picture {ck['picture']} {ck['hits']} of the {ck['places']} predicted places are also "
                            f"patches (random placement: {ck['random_mean_hits']} on average, chance {ck['chance'] * 100:.0f}%).")
            if len(row["seen_in_images"]) > 1:
                row["note"] += f" Both pictures together: chance {row['chance'] * 100:.1f}%."
        row["presumption"] = ("presumption" if row["chance"] < 0.10 and row["n_found_beyond_seeds"] >= 2 else
                              "weak presumption" if row["chance"] <= 0.30 and row["n_found_beyond_seeds"] >= 1 else "none")
        row["confidence"] = round(float(np.clip(1 - row["chance"] * 3, 0.05, 0.95)), 2)
        return jsonify(row)

    # ------------------------------------------------------------------ files
    @app.get("/api/areas/<a>/input/<path:f>")
    def input_file(a, f):
        return send_from_directory(os.path.join(adir(a), "input"), f)

    @app.get("/api/areas/<a>/result/<path:f>")
    def result_file(a, f):
        return send_from_directory(os.path.join(adir(a), "result"), f)

    @app.errorhandler(400)
    @app.errorhandler(404)
    def err(e):
        return jsonify({"error": getattr(e, "description", str(e))}), e.code

    return app


def serve(workspace, port=8765, open_browser=True):
    app = create_app(workspace)
    url = f"http://127.0.0.1:{port}/"
    print(f"sitecheck interface: {url}  (workspace: {os.path.abspath(workspace)})  Ctrl+C to stop")
    if open_browser:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
