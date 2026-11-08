"""Screen scale of a screenshot from the viewer's eye altitude.

Two different numbers are easy to mix up:
  * the IMAGERY resolution (what Airbus / Maxar captured: 30, 50, 70 cm per pixel). It says how
    much detail exists - nothing smaller than ~3 of those pixels can be seen at all;
  * the SCREEN scale (how many metres one pixel of your screenshot covers). It depends on how high
    the virtual camera was (Google Earth "eye altitude") and on the window size, not on the imagery.

Screen scale for a camera looking straight down:
    metres per pixel = 2 * altitude * tan(FOV / 2) / window width in pixels
Google Earth's default horizontal field of view is about 60 degrees; with a ~1920 px wide view this
is altitude / 1660 (e.g. 412 m -> 0.25 m/px, 1.7 km -> 1.0 m/px). Checked on the user's marked
foundations: with this scale their median size comes out 8-13 m in every area (312-473 m altitude).
"""
from __future__ import annotations

import math
import os
import re

FOV_DEG = 60.0
VIEW_WIDTH_PX = 1920


def mpp_from_altitude(altitude_m, view_width_px=VIEW_WIDTH_PX, fov_deg=FOV_DEG):
    return 2.0 * altitude_m * math.tan(math.radians(fov_deg) / 2) / view_width_px


def _metres(num, unit):
    v = float(num.replace(",", "."))
    unit = unit.lower()
    return v * 1000 if unit.startswith("k") else v / 100 if unit.startswith("c") else v


def parse_notes(text, names):
    """Read a notes .txt like
        for all height 473 m
        (for (notes,1) images 30 cm in each pixel )
        (for (2,3,4,5) images 50 cm in each pixel )
    or, per picture,
        (for All images 30 cm in each pixel )
        1) height 493 m
        7) height 1.34 km
    names: picture file names in the folder. Returns {file name: {"altitude_m", "imagery_m", "mpp"}}."""
    stems = {os.path.splitext(n)[0].lower(): n for n in names}
    out = {n: {} for n in names}
    num = r"(\d+(?:[.,]\d+)?)\s*(km|m|cm)\b"
    for line in text.splitlines():
        low = line.lower().strip()
        if not low:
            continue
        targets = None
        m_all = re.search(r"\bfor\s+all\b", low)
        m_list = re.search(r"for\s*\(([^)]*)\)", low)
        m_num = re.match(r"^\s*(\w+)\s*[).:-]\s", low)
        if m_list:
            keys = [k.strip() for k in m_list.group(1).split(",")]
            targets = [stems[k] for k in keys if k in stems]
        elif m_num and m_num.group(1) in stems:
            targets = [stems[m_num.group(1)]]
        elif m_all:
            targets = list(names)
        if not targets:
            continue
        h = re.search(r"(?:height|altitude|высота)\D*?" + num, low)
        res = re.search(num + r"\s*(?:in|per|на)?\s*(?:each|1|один)?\s*(?:pixel|px|пикс)", low)
        for t in targets:
            if h:
                out[t]["altitude_m"] = _metres(h.group(1), h.group(2))
            if res:
                out[t]["imagery_m"] = _metres(res.group(1), res.group(2))
    for n, d in out.items():
        if "altitude_m" in d:
            d["mpp"] = round(mpp_from_altitude(d["altitude_m"]), 4)
    return out


def read_folder(folder, names):
    """All .txt files in a folder, merged."""
    merged = {n: {} for n in names}
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(".txt"):
            try:
                with open(os.path.join(folder, f), encoding="utf-8", errors="ignore") as fh:
                    got = parse_notes(fh.read(), names)
            except OSError:
                continue
            for n, d in got.items():
                merged[n].update(d)
    return merged
