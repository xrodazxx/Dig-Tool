"""Finland: National Land Survey (Maanmittauslaitos) 2 m elevation model, from lidar.

Uses the NLS open-data OGC API Processes download service. It is free but needs a
personal API key: register at https://omatili.maanmittauslaitos.fi/ ("avoimet
rajapintapalvelut"), then either pass --nls-key KEY, or set NLS_API_KEY, or put
{"nls_api_key": "KEY"} into ~/.sitecheck.json.

The service is asynchronous: we start a job for a bounding box, wait for it, download
the result files (GeoTIFF or zip of GeoTIFF tiles, EPSG:3067), join and cut them.
The process and its inputs are discovered from the service itself, so small API
changes don't break the tool.
"""
from __future__ import annotations

import io
import json
import os
import time
import zipfile

from .. import proj
from ..raster import crop_around, interpolation_check, load_dem, mosaic, save_geotiff
from .common import NoData, bbox, get, is_tiff, post

BASES = ["https://avoin-paikkatieto.maanmittauslaitos.fi/tiedostopalvelu/ogcproc/v1",
         "https://beta-paikkatieto.maanmittauslaitos.fi/tiedostopalvelu/ogcproc/v1"]
CRS_3067 = "http://www.opengis.net/def/crs/EPSG/0/3067"


def api_key(explicit=None):
    if explicit:
        return explicit
    if os.environ.get("NLS_API_KEY"):
        return os.environ["NLS_API_KEY"]
    cfg = os.path.expanduser("~/.sitecheck.json")
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as fh:
            return json.load(fh).get("nls_api_key")
    return None


def _auth(key):
    # NLS accepts the key as 'api-key' query parameter; basic auth (key as user) is the
    # documented alternative for some services - we send both.
    return {"params": {"api-key": key}, "auth": (key, "")}


def _pick_process(processes):
    ids = [p.get("id", "") for p in processes]
    score = lambda i: (("korkeusmalli" in i) * 4 + ("2m" in i or "2_m" in i) * 2 + ("bbox" in i) * 1)
    best = max(ids, key=score, default="")
    if score(best) < 6:
        raise RuntimeError("NLS: no 2 m elevation process found. Available: " + ", ".join(ids[:20]))
    return best


def _build_inputs(desc, e0, n0, e1, n1):
    """Fill the process inputs from its description."""
    inputs = {}
    for name, spec in (desc.get("inputs") or {}).items():
        low = name.lower()
        schema = spec.get("schema", {}) if isinstance(spec, dict) else {}
        enum = schema.get("enum") or (schema.get("items") or {}).get("enum")
        if "bbox" in low or "boundingbox" in low:
            if schema.get("type") == "string":
                inputs[name] = f"{e0},{n0},{e1},{n1}"
            else:
                inputs[name] = {"bbox": [e0, n0, e1, n1], "crs": CRS_3067}
        elif "format" in low and enum:
            tif = [v for v in enum if "tif" in str(v).lower()]
            inputs[name] = tif[0] if tif else enum[0]
        elif ("crs" in low or "srs" in low) and enum:
            pick = [v for v in enum if "3067" in str(v)]
            inputs[name] = pick[0] if pick else enum[0]
        elif enum and (spec.get("minOccurs", 1) if isinstance(spec, dict) else 1) > 0:
            inputs[name] = enum[0]
    if not any("bbox" in k.lower() or "boundingbox" in k.lower() for k in inputs):
        raise RuntimeError("NLS: process has no bounding-box input: " + ", ".join(desc.get("inputs", {})))
    return inputs


def _hrefs(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "href" and isinstance(v, str):
                yield v
            else:
                yield from _hrefs(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _hrefs(v)
    elif isinstance(obj, str) and obj.startswith("http"):
        yield obj


def nls_finland(lat, lon, radius_m, out_path, key=None, wait_s=600, log=print):
    key = api_key(key)
    if not key:
        raise RuntimeError("Finland 2 m DEM needs a free NLS API key (see `sitecheck fetch --help`).")
    w, s, e, n = bbox(lat, lon, radius_m + 20)
    xs, ys = zip(*(proj.forward(3067, x, y) for x, y in ((w, s), (e, s), (w, n), (e, n))))
    e0, n0, e1, n1 = min(xs), min(ys), max(xs), max(ys)
    a = _auth(key)
    last_err = None
    for base in BASES:
        try:
            r = get(base + "/processes", retries=2, **a)
            if r.status_code in (401, 403):
                raise RuntimeError("NLS rejected the API key (HTTP %d)" % r.status_code)
            pid = _pick_process(r.json().get("processes", []))
            desc = get(f"{base}/processes/{pid}", **a).json()
            body = {"inputs": _build_inputs(desc, e0, n0, e1, n1)}
            log(f"  NLS job: {pid}")
            r = post(f"{base}/processes/{pid}/execution", json=body,
                     headers={"Prefer": "respond-async"}, **a)
            if r.status_code >= 400:
                raise RuntimeError(f"NLS execution failed (HTTP {r.status_code}): {r.text[:300]}")
            job_url = r.headers.get("Location")
            job = r.json() if r.content and r.headers.get("Content-Type", "").startswith("application/json") else {}
            if not job_url:
                jid = job.get("jobID") or job.get("jobId") or job.get("id")
                job_url = f"{base}/jobs/{jid}" if jid else None
            results = None
            if job_url:
                t0 = time.time()
                while time.time() - t0 < wait_s:
                    st = get(job_url, **a).json()
                    status = str(st.get("status", "")).lower()
                    if status in ("successful", "succeeded", "success", "finished"):
                        break
                    if status in ("failed", "dismissed"):
                        raise RuntimeError(f"NLS job {status}: {st.get('message', '')}")
                    time.sleep(5)
                else:
                    raise RuntimeError("NLS job did not finish in time; try again later")
                results = get(job_url.rstrip("/") + "/results", **a).json()
            else:
                results = job
            files = [h for h in _hrefs(results) if h.split("?")[0].lower().endswith((".tif", ".tiff", ".zip"))]
            if not files:
                raise NoData("NLS returned no files for this area")
            tiles = _download_tiles(files, os.path.dirname(out_path) or ".", a)
            dem = mosaic([load_dem(t) for t in tiles])
            sub, _ = crop_around(dem, lon, lat, radius_m + 10)
            save_geotiff(sub.z, sub.x0, sub.y0, sub.dx, sub.dy, out_path, 3067)
            return {"file": out_path, "label": "NLS Finland elevation model 2 m (lidar)", "cell_m": 2.0,
                    "interpolation": interpolation_check(sub.z),
                    "licence": "CC BY 4.0, National Land Survey of Finland"}
        except NoData:
            raise
        except Exception as ex:  # noqa: BLE001 - try next base URL
            last_err = ex
    raise RuntimeError(f"NLS download failed: {last_err}")


def _download_tiles(urls, folder, a):
    out = []
    for i, u in enumerate(urls):
        r = get(u, timeout=600, **a)
        if r.status_code != 200:
            continue
        if is_tiff(r.content):
            p = os.path.join(folder, f"nls_tile_{i}.tif")
            with open(p, "wb") as fh:
                fh.write(r.content)
            out.append(p)
        elif r.content[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                for name in zf.namelist():
                    if name.lower().endswith((".tif", ".tiff")):
                        p = os.path.join(folder, f"nls_tile_{i}_{os.path.basename(name)}")
                        with open(p, "wb") as fh:
                            fh.write(zf.read(name))
                        out.append(p)
    if not out:
        raise NoData("NLS result contained no GeoTIFF")
    return out
