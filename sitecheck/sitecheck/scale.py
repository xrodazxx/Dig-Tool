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


PROVIDERS = ("cnes/airbus", "airbus", "maxar", "digitalglobe", "landsat", "copernicus")


def _date(txt):
    """'6.25.2013' (month.day.year), '2013-06-25' or '2013-06' -> ISO date string."""
    m = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b", txt)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        mo, d = (a, b) if a <= 12 else (b, a)          # month first; swap if that cannot be
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    m = re.search(r"\b(\d{4})-(\d{1,2})(?:-(\d{1,2}))?\b", txt)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3) or 15):02d}"
    return None


def guess_imagery_m(provider, date):
    """Rough ground resolution of what Google Earth shows, when the notes do not give it:
    Airbus Pleiades Neo (2021+) and Maxar WorldView-3/Legion (2015+) ~0.3-0.5 m, CNES/Airbus Pleiades
    0.5 m, Maxar 2008-2014 ~0.5 m, older (QuickBird) 0.6-0.7 m. An estimate, marked as such."""
    if not provider or not date:
        return None
    y = int(date[:4])
    p = provider.lower()
    if "cnes" in p:
        return 0.5
    if "airbus" in p:
        return 0.3 if y >= 2021 else 0.5
    if "maxar" in p or "digitalglobe" in p:
        return 0.3 if y >= 2015 else 0.5 if y >= 2008 else 0.65
    return None


def sun_position(lat, date, solar_hour=10.8):
    """(azimuth from north clockwise, elevation), degrees, at the usual pass time of the imaging
    satellites (Maxar, Airbus: about 10:30-11:00 local solar time). Good to a few degrees - enough
    to know which side of a hollow is lit."""
    try:
        y, mo, d = (int(x) for x in date.split("-"))
    except (ValueError, AttributeError):
        return None, None
    doy = (mo - 1) * 30.4 + d
    decl = math.radians(23.44) * math.sin(2 * math.pi * (doy - 81) / 365.0)
    ha = math.radians(15.0 * (solar_hour - 12.0))
    la = math.radians(lat)
    s = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    el = math.degrees(math.asin(max(-1.0, min(1.0, s))))
    az = (math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(la) - math.tan(decl) * math.cos(la))) + 180) % 360
    return round(az, 1), round(el, 1)


def sun_elevation(lat, date, solar_hour=10.8):
    return sun_position(lat, date, solar_hour)[1]


def parse_notes(text, names, view_width_px=None):
    """Read an area's notes .txt. Understood lines (order and brackets do not matter much):

        for all height 473 m                      eye altitude of every picture
        7) height 1.34 km                         ... or per picture
        window 1920 px                            width of the map window (default 1920)
        coord (54 19 44 N 50 19 38 E)             place (degrees minutes seconds, or decimal)
        (for (notes,1) Maxar 6.25.2013 )          who took picture 1 (and the marked copy) and when (M.D.YYYY)
        (for (2,3) images 50 cm in each pixel )   imagery resolution, if known
        visible: 1,2                              how well the marked things show in each picture
        doubtful: 3
        not visible: 4

    names: picture file names in the folder. Returns {file name: {...}} plus "_area": {lat, lon, ...}."""
    stems = {os.path.splitext(n)[0].lower(): n for n in names}
    out = {n: {} for n in names}
    area = {}
    num = r"(\d+(?:[.,]\d+)?)\s*(km|m|cm)\b"
    view_w = view_width_px or VIEW_WIDTH_PX

    def keys_to_files(txt):
        return [stems[k] for k in re.findall(r"[a-z0-9_]+", txt.lower()) if k in stems]

    for line in text.replace("\ufeff", "").splitlines():       # (Notepad's invisible BOM)
        low = line.lower().strip()
        if not low:
            continue
        # --- visibility ------------------------------------------------------------------
        mv = re.match(r"^\s*(not\s*visible|invisible|doubtful|questionable|visible)\s*:(.*)$", low)
        if mv:
            cls = mv.group(1)
            cls = "not visible" if cls.startswith(("not", "inv")) else "doubtful" if cls[0] in "dq" else "visible"
            rank = {"visible": 0, "doubtful": 1, "not visible": 2}
            for f in keys_to_files(mv.group(2)):
                old = out[f].get("visibility")
                if old is None or rank[cls] > rank[old]:       # listed twice: the more cautious one
                    out[f]["visibility"] = cls
            continue
        # --- coordinates -----------------------------------------------------------------
        mc = re.match(r"^\s*\(?\s*[a-z]{0,4}(?:coords?|cords?|coordinates?)\b[\s:(]*(.*)$", low)
        if mc:
            ll = _latlon(mc.group(1))
            if ll:
                area["lat"], area["lon"] = ll
            continue
        mg = re.match(r"^\s*\(?\s*(?:elev(?:ation)?|ground)\D{0,10}?(\d+(?:[.,]\d+)?)\s*m\b", low)
        if mg:
            area["ground_m"] = _metres(mg.group(1), "m")
            continue
        mw = re.search(r"\bwindow\D*(\d{3,4})\s*(?:px)?", low)
        if mw:
            view_w = int(mw.group(1))
            area["view_width_px"] = view_w
            continue
        ms = re.search(r"\bsize\b.*?(\d{3,4})\s*[x×х]\s*(\d{3,4})", low)
        if ms:
            area["stated_size"] = [int(ms.group(1)), int(ms.group(2))]
            continue
        # --- which pictures does the line speak about --------------------------------------
        targets = None
        m_all = re.search(r"\bfor\s+all\b", low)
        m_list = re.search(r"for\s*\(\s*([a-z0-9_]+(?:\s*,\s*[a-z0-9_]+)*)", low)
        m_num = re.match(r"^\s*(\w+)\s*[).:-]\s", low)
        if m_all:
            targets = list(names)
        elif m_list:
            targets = keys_to_files(m_list.group(1))
        elif m_num and m_num.group(1) in stems:
            targets = [stems[m_num.group(1)]]
        if not targets:
            continue
        h = re.search(r"(?:height|altitude|высота)\D*?" + num, low)
        res = re.search(num + r"\s*(?:in|per|на)?\s*(?:each|1|один)?\s*(?:pixel|px|пикс)", low)
        prov = next((p for p in PROVIDERS if p in low), None)
        date = _date(low) if not m_all else None
        for t in targets:
            if h:
                out[t]["altitude_m"] = _metres(h.group(1), h.group(2))
            if res:
                out[t]["imagery_m"] = _metres(res.group(1), res.group(2))
            if prov:
                out[t]["provider"] = "CNES/Airbus" if "cnes" in prov else prov.capitalize()
            if date:
                out[t]["date"] = date
    for n, d in out.items():
        if "altitude_m" in d:
            d["mpp"] = round(mpp_from_altitude(d["altitude_m"], view_w), 4)
        if "imagery_m" not in d:
            g = guess_imagery_m(d.get("provider"), d.get("date"))
            if g:
                d["imagery_m"], d["imagery_m_estimated"] = g, True
        if d.get("date") and area.get("lat") is not None:
            d["sun_azimuth"], d["sun_elevation"] = sun_position(area["lat"], d["date"])
    out["_area"] = area
    return out


def _latlon(txt):
    """'54 19 44 N 50 19 38 E', '51 36 N 43 44 E', '54.33 50.32', '54°19'44" N ...' -> (lat, lon)."""
    t = txt.replace(",", " ").replace("с", "n").replace("в", "e").replace("ю", "s").replace("з", "w")
    parts = re.findall(r"((?:\d+(?:\.\d+)?[\s°'\"′″]*){1,3})\s*([nsew])", t)
    vals = []
    for nums, hemi in parts:
        v = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", nums)]
        deg = v[0] + (v[1] / 60 if len(v) > 1 else 0) + (v[2] / 3600 if len(v) > 2 else 0)
        vals.append((-deg if hemi in "sw" else deg, hemi))
    if len(vals) >= 2:
        lat = next((v for v, h in vals if h in "ns"), None)
        lon = next((v for v, h in vals if h in "ew"), None)
        if lat is not None and lon is not None:
            return round(lat, 5), round(lon, 5)
    v = [float(x) for x in re.findall(r"-?\d+\.\d+", t)]
    if len(v) == 2 and abs(v[0]) <= 90 and abs(v[1]) <= 180:
        return v[0], v[1]
    return None


def _pack_info(folder):
    """Lines in a .txt one or two folders up (the pack's General info) that hold for every area:
        window 1749 px                    width of the map view in screen pixels
        View distance 400 m / height 400 m   eye altitude of every picture (unless an area says otherwise)
        elev 150 m                        ground height above sea level (eye altitude is above sea level)"""
    d = os.path.abspath(folder)
    out = {}
    for _ in range(2):
        d = os.path.dirname(d)
        try:
            files = sorted(f for f in os.listdir(d) if f.lower().endswith(".txt"))
        except OSError:
            break
        for f in files:
            try:
                with open(os.path.join(d, f), encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read().lower()
            except OSError:
                continue
            m = re.search(r"^\s*(?:map\s+)?window\D{0,20}(\d{3,4})", txt, re.M)
            if m and "window" not in out:
                out["window"] = int(m.group(1))
            m = re.search(r"(?:view\s*distance|eye\s*alt(?:itude)?|height|altitude)\D{0,15}?"
                          r"(\d+(?:[.,]\d+)?)\s*(km|m)\b", txt)
            if m and "altitude_m" not in out:
                out["altitude_m"] = _metres(m.group(1), m.group(2))
            m = re.search(r"(?:\belev(?:ation)?|ground)\D{0,10}?(\d+(?:[.,]\d+)?)\s*(m)\b", txt)
            if m and "ground_m" not in out:
                out["ground_m"] = _metres(m.group(1), m.group(2))
        if out:
            break
    return out


def _pack_window(folder):
    return _pack_info(folder).get("window")


def read_folder(folder, names):
    """All .txt files in a folder, merged. "_area" holds lat/lon and other whole-area facts.
    Pack-wide lines (window, eye altitude, ground height) fill in what an area does not say."""
    merged = {n: {} for n in names}
    merged["_area"] = {}
    pk = _pack_info(folder)
    vw = pk.get("window")
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(".txt"):
            try:
                with open(os.path.join(folder, f), encoding="utf-8", errors="ignore") as fh:
                    got = parse_notes(fh.read(), names, vw)
            except OSError:
                continue
            for n, d in got.items():
                merged.setdefault(n, {}).update(d)
    area = merged["_area"]
    if vw and "view_width_px" not in area:
        area["view_width_px"] = vw
    ground = area.get("ground_m", pk.get("ground_m", 0.0))
    for n in names:
        d = merged[n]
        if "altitude_m" not in d and "altitude_m" in pk:
            d["altitude_m"] = pk["altitude_m"]
        if "altitude_m" in d:          # eye altitude is above sea level: the camera is that minus the ground
            d["mpp"] = round(mpp_from_altitude(max(d["altitude_m"] - ground, 30.0),
                                               area.get("view_width_px", VIEW_WIDTH_PX)), 4)
    return merged


def looks_like_notes(text):
    """A notes file (heights, picture dates, visibility) rather than a bare coordinates file."""
    return bool(re.search(r"height|altitude|for\s*\(|visible|doubtful|pixel", text.lower()))
