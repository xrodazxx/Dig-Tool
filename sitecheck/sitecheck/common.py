"""Shared helpers for the optional download step."""
from __future__ import annotations

import math
import time

import requests

UA = {"User-Agent": "sitecheck/0.2 (offline terrain analysis tool)"}


def bbox(lat, lon, r):
    """(west, south, east, north) of a square +/- r meters around the point."""
    dlat = r / 110_574
    dlon = r / (111_320 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def get(url, retries=3, **kw):
    kw.setdefault("timeout", 120)
    headers = {**UA, **kw.pop("headers", {})}
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=headers, **kw)
            if r.status_code in (429, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} from {url}")
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as ex:
            last = ex
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"cannot reach {url.split('?')[0]}: {last}")


def post(url, retries=2, **kw):
    kw.setdefault("timeout", 120)
    headers = {**UA, **kw.pop("headers", {})}
    last = None
    for i in range(retries):
        try:
            return requests.post(url, headers=headers, **kw)
        except (requests.ConnectionError, requests.Timeout) as ex:
            last = ex
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"cannot reach {url.split('?')[0]}: {last}")


def is_tiff(b: bytes) -> bool:
    return b[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


class NoData(RuntimeError):
    """The source answered but has no elevation for this spot."""
